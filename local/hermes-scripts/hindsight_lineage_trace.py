#!/usr/bin/env python3
"""Trace Hindsight high-level observations/facts/documents down to low-level evidence and raw spans.

Read-only: no Hindsight writes, no DB writes, no LLM calls. Hindsight data-plane
lookups go through official APIs; local SQLite is only used for Hermes raw-span
provenance.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hindsight_conflict_core as core
from hindsight_native_client import DEFAULT_API, HindsightNativeClient

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_STATE_DB = HERMES_HOME / "state.db"
DEFAULT_OFFLINE_ROOT = HERMES_HOME / "hindsight" / "offline_reflect"
DEFAULT_CARDS_ROOT = DEFAULT_OFFLINE_ROOT / "v2_cards"
DEFAULT_OUTPUT_DIR = DEFAULT_OFFLINE_ROOT / "lineage_traces"


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_observations(cards_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    index_path = cards_root / "observations_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("insight"):
                out.append(obj)
    if out:
        return out
    for p in sorted(list((cards_root / "topics").glob("*.json")) + list((cards_root / "global").glob("*.json"))):
        card = load_json(p, {}) or {}
        for obs in card.get("canonical_observations") or []:
            if isinstance(obs, dict):
                row = dict(obs)
                row.setdefault("topic", card.get("topic"))
                out.append(row)
    return out


def find_observation(target: str, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not target:
        return None
    for obs in observations:
        if str(obs.get("id") or "") == target:
            return obs
    low = target.lower()
    best = None
    for obs in observations:
        hay = " ".join([str(obs.get("id") or ""), str(obs.get("insight") or ""), str(obs.get("topic") or ""), " ".join(map(str, obs.get("tags") or []))]).lower()
        if low in hay:
            best = obs
            break
    return best


def document_row(document_id: str, bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None) -> dict[str, Any] | None:
    try:
        row = (client or HindsightNativeClient(api=api, bank=bank)).get_document(document_id)
    except Exception:
        return None
    original_text = str(row.get("original_text") or "")
    doc_id = str(row.get("id") or document_id)
    metadata = row.get("document_metadata") or row.get("metadata") or {}
    return {
        "id": doc_id,
        "layer": classify_doc(doc_id),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "metadata": metadata,
        "text_preview": core.redact(original_text.replace("\n", " "), 800),
        "contamination": core.detect_contamination(original_text),
        "source_sessions": core.extract_source_sessions(original_text)[:100],
        "numeric_tokens": core.numeric_tokens(original_text)[:30],
        "source": "official_api_document_get",
    }


def memory_row(memory_id: str, bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None) -> dict[str, Any] | None:
    try:
        row = (client or HindsightNativeClient(api=api, bank=bank)).get_memory(memory_id)
    except Exception:
        return None
    text = str(row.get("text") or "")
    return {
        "id": row.get("id"),
        "document_id": row.get("document_id"),
        "fact_type": row.get("type") or row.get("fact_type"),
        "text_preview": core.redact(text.replace("\n", " "), 1000),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at") or row.get("date") or row.get("mentioned_at"),
        "source_memory_ids": [str(x) for x in (row.get("source_memory_ids") or row.get("source_fact_ids") or [])],
        "contamination": core.detect_contamination(text),
        "numeric_tokens": core.numeric_tokens(text),
        "source": "official_api_memory_get",
    }


def load_inline_json(text: str) -> Any:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {"_raw": core.redact(text, 500)}


def classify_doc(document_id: str) -> str:
    if document_id.startswith("hermes-offline-canonical::"):
        return "canonical"
    if document_id.startswith("hermes-offline-consolidation::weekly::"):
        return "offline_weekly"
    if document_id.startswith("hermes-offline-consolidation::daily::"):
        return "offline_daily"
    if document_id.startswith("hermes-sqlite::"):
        return "sqlite_import"
    return "other"


def file_artifact(path_text: str) -> dict[str, Any] | None:
    if not path_text.startswith("/") and not path_text.startswith("~"):
        return None
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    data = load_json(path, None) if path.suffix == ".json" else None
    sibling_json = path.with_suffix(".json") if path.suffix != ".json" else None
    if data is None and sibling_json and sibling_json.exists():
        data = load_json(sibling_json, None)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "suffix": path.suffix,
    }
    if isinstance(data, dict):
        unit = data.get("unit") or {}
        result.update({
            "document_id": data.get("document_id"),
            "unit": unit,
            "llm_json_keys": list((data.get("llm_json") or {}).keys())[:30],
            "source_ids": (unit.get("source_ids") or [])[:80],
            "raw_text_contamination": core.detect_contamination(str(data.get("raw_text") or "")),
            "raw_text_preview": core.redact(str(data.get("raw_text") or "").replace("\n", " "), 800),
        })
        if sibling_json and sibling_json.exists():
            result["json_sidecar"] = str(sibling_json)
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:3000]
            result.update({
                "text_preview": core.redact(text.replace("\n", " "), 800),
                "contamination": core.detect_contamination(text),
            })
        except Exception as e:
            result["read_error"] = repr(e)
    return result


def raw_session_summary(session_id: str, db_path: Path) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        sess = con.execute("SELECT id, source, model, started_at, ended_at, title, message_count, tool_call_count, input_tokens, output_tokens FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not sess:
            return None
        rows = con.execute(
            "SELECT role, timestamp, length(coalesce(content,'')) AS chars, substr(coalesce(content,''),1,500) AS preview FROM messages WHERE session_id=? ORDER BY timestamp LIMIT 20",
            (session_id,),
        ).fetchall()
        scan_rows = con.execute(
            "SELECT role, timestamp, coalesce(content,'') AS content FROM messages WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        total = con.execute("SELECT count(*), sum(length(coalesce(content,''))) FROM messages WHERE session_id=?", (session_id,)).fetchone()
    finally:
        con.close()
    messages = []
    contamination_hits = []
    hit_messages = []
    for r in scan_rows:
        content = str(r["content"] or "")
        hits = core.detect_contamination(content)
        if hits:
            contamination_hits.extend(hits)
            hit_messages.append({
                "role": r["role"],
                "timestamp": r["timestamp"],
                "hits": hits[:5],
                "preview": core.redact(content.replace("\n", " "), 500),
            })
    for r in rows:
        preview = str(r["preview"] or "")
        hits = core.detect_contamination(preview)
        messages.append({
            "role": r["role"],
            "timestamp": r["timestamp"],
            "chars": r["chars"],
            "preview": core.redact(preview.replace("\n", " "), 500),
            "contamination": hits,
        })
    return {
        "id": sess["id"],
        "source": sess["source"],
        "model": sess["model"],
        "started_at": sess["started_at"],
        "ended_at": sess["ended_at"],
        "title": sess["title"],
        "message_count": total[0],
        "content_chars": total[1],
        "tool_call_count": sess["tool_call_count"],
        "input_tokens": sess["input_tokens"],
        "output_tokens": sess["output_tokens"],
        "message_samples": messages,
        "contamination": contamination_hits[:20],
        "contamination_summary": {
            "scanned_messages": len(scan_rows),
            "hit_messages": len(hit_messages),
            "hit_samples": hit_messages[:10],
        },
    }


def trace_target(
    target: str,
    *,
    bank: str,
    cards_root: Path,
    state_db: Path,
    max_depth: int = 4,
    api: str = DEFAULT_API,
    client: HindsightNativeClient | None = None,
) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    observations = load_observations(cards_root)
    obs = find_observation(target, observations)
    trace: dict[str, Any] = {
        "target": target,
        "matched_observation": obs,
        "layers": [],
        "raw_sessions": [],
        "unresolved_refs": [],
        "repair_hints": [],
    }
    pending_refs: list[str] = []
    if obs:
        trace["layers"].append({
            "kind": "local_canonical_observation",
            "id": obs.get("id"),
            "topic": obs.get("topic"),
            "type": obs.get("type"),
            "insight": core.redact(str(obs.get("insight") or ""), 1000),
            "source_documents": obs.get("source_documents") or [],
            "evidence_ids": obs.get("evidence_ids") or [],
            "numeric_tokens": core.numeric_tokens(str(obs.get("insight") or "")),
            "contamination": core.detect_contamination(core.observation_text(obs)),
        })
        pending_refs.extend(core.source_refs(obs))
    else:
        pending_refs.append(target)

    seen_refs = set()
    raw_session_ids = set()
    for depth in range(max_depth):
        if not pending_refs:
            break
        next_refs: list[str] = []
        for ref in pending_refs:
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            ref = ref.strip()
            if not ref:
                continue
            # UUID memory facts.
            uuids = core.UUID_RE.findall(ref)
            if uuids:
                for uid in uuids:
                    row = memory_row(uid, bank, api=api, client=client)
                    if row:
                        row["kind"] = "memory_unit"
                        row["depth"] = depth
                        trace["layers"].append(row)
                        if row.get("document_id"):
                            next_refs.append(str(row["document_id"]))
                        next_refs.extend(map(str, row.get("source_memory_ids") or []))
                    else:
                        trace["unresolved_refs"].append({"ref": uid, "kind": "memory_unit", "depth": depth})
                continue
            # Hindsight document ids.
            if ref.startswith("hermes-"):
                doc = document_row(ref, bank, api=api, client=client)
                if doc:
                    doc["kind"] = "document"
                    doc["depth"] = depth
                    trace["layers"].append(doc)
                    for sess in doc.get("source_sessions") or []:
                        sid = sess.get("id")
                        if sid:
                            raw_session_ids.add(str(sid))
                    # If this document is itself a daily/weekly consolidation, try local file via document_id not always possible; evidence_ids often include file paths.
                else:
                    trace["unresolved_refs"].append({"ref": ref, "kind": "document", "depth": depth})
                continue
            # Local artifact paths.
            artifact = file_artifact(ref)
            if artifact:
                artifact["kind"] = "local_file"
                artifact["depth"] = depth
                trace["layers"].append(artifact)
                unit = artifact.get("unit") or {}
                next_refs.extend(map(str, artifact.get("source_ids") or []))
                if artifact.get("document_id"):
                    next_refs.append(str(artifact["document_id"]))
                continue
            trace["unresolved_refs"].append({"ref": ref, "kind": "unknown", "depth": depth})
        pending_refs = next_refs

    for sid in sorted(raw_session_ids):
        raw = raw_session_summary(sid, state_db)
        if raw:
            trace["raw_sessions"].append(raw)
        else:
            trace["unresolved_refs"].append({"ref": sid, "kind": "raw_session", "depth": "raw"})

    if not trace["matched_observation"] and not trace["layers"]:
        trace["repair_hints"].append("No matching observation/document/fact found; verify target id/text or run conflict audit first.")
    if any(layer.get("contamination") for layer in trace["layers"]):
        trace["repair_hints"].append("Contamination detected in lineage; quarantine candidate before considering publish.")
    if trace["unresolved_refs"]:
        trace["repair_hints"].append("Some references are unresolved; repair lineage or keep local-only until traceable.")
    if trace["raw_sessions"]:
        trace["repair_hints"].append("Raw sessions recovered; inspect message_samples before re-retain/delete decisions.")
    return trace


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Hindsight Lineage Trace", ""]
    lines.append(f"generated_at: {report.get('generated_at')}")
    lines.append(f"target: {report.get('target')}")
    lines.append(f"layers: {len(report.get('layers') or [])}")
    lines.append(f"raw_sessions: {len(report.get('raw_sessions') or [])}")
    lines.append(f"unresolved_refs: {len(report.get('unresolved_refs') or [])}")
    lines.append("")
    lines.append("## Repair Hints")
    for h in report.get("repair_hints") or []:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("## Layer Trace")
    for item in report.get("layers") or []:
        lines.append(f"- depth={item.get('depth')} kind={item.get('kind')} id={item.get('id') or item.get('path') or item.get('document_id')}")
        preview = item.get("insight") or item.get("text_preview") or item.get("raw_text_preview")
        if preview:
            lines.append(f"  preview: {core.redact(str(preview), 300)}")
        if item.get("contamination"):
            lines.append(f"  contamination: {item.get('contamination')}")
    lines.append("")
    lines.append("## Raw Sessions")
    for sess in report.get("raw_sessions") or []:
        lines.append(f"- {sess.get('id')} messages={sess.get('message_count')} chars={sess.get('content_chars')} title={sess.get('title')}")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace Hindsight lineage to facts/documents/raw sessions")
    ap.add_argument("--target", required=True, help="Observation id, fact UUID, document id, or text substring")
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT))
    ap.add_argument("--state-db", default=str(DEFAULT_STATE_DB))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = trace_target(
        args.target,
        bank=args.bank,
        cards_root=Path(args.cards_root).expanduser(),
        state_db=Path(args.state_db).expanduser(),
        max_depth=args.max_depth,
        api=args.api,
    )
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["safety"] = ["read_only", "no_llm_calls", "no_db_writes", "destructive_actions_require_user_confirmation"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.target)[:60].strip("-._") or "target"
    json_path = output_dir / f"lineage-{slug}-{ts}.json"
    md_path = output_dir / f"lineage-{slug}-{ts}.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        print(f"saved_json: {json_path}")
        print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
