#!/usr/bin/env python3
"""Discard/quarantine snapshot manager for Hindsight production mutations.

This script is intentionally discard-first and non-mutating by default. It saves
local JSON/JSONL snapshots of affected documents/facts/observations before any
separate production delete/replace/clear operation is allowed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DEFAULT_API, HindsightNativeClient  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path.home() / ".hermes" / "hindsight" / "discard"
DEFAULT_DISCARD_BANK = "hermes_discard"
SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[:=]\s*)([^\s,'\"]+)")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text.strip())
    text = text.strip("-._")
    return text[:120] or "case"


def redact(obj: Any) -> Any:
    if isinstance(obj, str):
        return SECRET_RE.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", obj)
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if re.search(r"(?i)(api[_-]?key|token|secret|password|authorization)", str(k)) else redact(v)) for k, v in obj.items()}
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(redact(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def memory_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("fact_type") or "")


def memory_document_id(row: dict[str, Any]) -> str:
    return str(row.get("document_id") or "")


def source_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("source_memory_ids") or row.get("source_fact_ids") or []
    if isinstance(ids, list):
        return [str(x) for x in ids if x]
    return []


def all_memories(client: Any) -> list[dict[str, Any]]:
    return [m for m in client.iter_memories(types=None, max_items=None) if isinstance(m, dict)]


def snapshot_document(*, bank: str, document_id: str, case_id: str, output_root: str | Path = DEFAULT_OUTPUT_ROOT, client: Any | None = None, api: str = DEFAULT_API, reason: str = "", discard_bank: str = DEFAULT_DISCARD_BANK) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank, timeout=60)
    output_root = Path(output_root)
    case_dir = output_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_name(case_id)}"
    document = client.get_document(document_id)
    memories = all_memories(client)
    facts = [m for m in memories if memory_document_id(m) == document_id and memory_type(m) != "observation"]
    fact_ids = {str(m.get("id")) for m in facts if m.get("id")}
    derived_observations = [m for m in memories if memory_type(m) == "observation" and fact_ids.intersection(source_ids(m))]

    write_json(case_dir / "document.json", document)
    write_jsonl(case_dir / "facts.jsonl", facts)
    write_jsonl(case_dir / "derived_observations.jsonl", derived_observations)
    manifest = {
        "case_id": case_id,
        "created_at": now_iso(),
        "operation": "snapshot_document",
        "source_bank": bank,
        "discard_bank": discard_bank,
        "document_id": document_id,
        "reason": reason,
        "counts": {
            "documents": 1 if document else 0,
            "facts": len(facts),
            "derived_observations": len(derived_observations),
        },
        "files": {
            "document": "document.json",
            "facts": "facts.jsonl",
            "derived_observations": "derived_observations.jsonl",
        },
        "mutation_allowed": False,
        "note": "Snapshot only. Production mutation requires separate backup, verification, and explicit confirmation.",
    }
    write_json(case_dir / "manifest.json", manifest)
    return {"dry_run": True, "case_dir": str(case_dir), "document_id": document_id, "counts": manifest["counts"]}


def snapshot_observations_for_memory(*, bank: str, memory_id: str, case_id: str, output_root: str | Path = DEFAULT_OUTPUT_ROOT, client: Any | None = None, api: str = DEFAULT_API, reason: str = "", discard_bank: str = DEFAULT_DISCARD_BANK) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank, timeout=60)
    output_root = Path(output_root)
    case_dir = output_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_name(case_id)}"
    memory = client.get_memory(memory_id)
    memories = all_memories(client)
    observations = [m for m in memories if memory_type(m) == "observation" and memory_id in source_ids(m)]
    write_json(case_dir / "source_memory.json", memory)
    write_jsonl(case_dir / "derived_observations.jsonl", observations)
    manifest = {
        "case_id": case_id,
        "created_at": now_iso(),
        "operation": "snapshot_observations",
        "source_bank": bank,
        "discard_bank": discard_bank,
        "memory_id": memory_id,
        "reason": reason,
        "counts": {"source_memories": 1 if memory else 0, "derived_observations": len(observations)},
        "files": {"source_memory": "source_memory.json", "derived_observations": "derived_observations.jsonl"},
        "mutation_allowed": False,
        "note": "Snapshot only. Clearing observations requires separate backup, verification, and explicit confirmation.",
    }
    write_json(case_dir / "manifest.json", manifest)
    return {"dry_run": True, "case_dir": str(case_dir), "memory_id": memory_id, "counts": manifest["counts"]}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def verify_case(case_dir: str | Path) -> dict[str, Any]:
    case_dir = Path(case_dir)
    errors: list[str] = []
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "case_dir": str(case_dir), "errors": ["manifest.json missing"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "case_dir": str(case_dir), "errors": [f"manifest unreadable: {e}"]}
    counts = manifest.get("counts") or {}
    files = manifest.get("files") or {}
    if counts.get("documents", 0):
        doc_file = case_dir / str(files.get("document", "document.json"))
        if not doc_file.exists():
            errors.append("document snapshot missing")
    if "facts" in counts:
        n = count_jsonl(case_dir / str(files.get("facts", "facts.jsonl")))
        if n != int(counts.get("facts") or 0):
            errors.append(f"facts count mismatch: expected {counts.get('facts')} got {n}")
    if "derived_observations" in counts:
        n = count_jsonl(case_dir / str(files.get("derived_observations", "derived_observations.jsonl")))
        if n != int(counts.get("derived_observations") or 0):
            errors.append(f"derived_observations count mismatch: expected {counts.get('derived_observations')} got {n}")
    if manifest.get("mutation_allowed") is not False:
        errors.append("mutation_allowed must be false for discard snapshots")
    return {"ok": not errors, "case_dir": str(case_dir), "case_id": manifest.get("case_id"), "errors": errors, "counts": counts}


def list_cases(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    cases: list[dict[str, Any]] = []
    if not output_root.exists():
        return cases
    for manifest_path in sorted(output_root.glob("*/manifest.json"), reverse=True):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"case_id": manifest_path.parent.name, "unreadable": True}
        data["case_dir"] = str(manifest_path.parent)
        cases.append(data)
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discard-first Hindsight snapshot manager. No production mutation by default.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    snap_doc = sub.add_parser("snapshot-document")
    snap_doc.add_argument("--bank", required=True)
    snap_doc.add_argument("--document-id", required=True)
    snap_doc.add_argument("--case-id", required=True)
    snap_doc.add_argument("--reason", default="")
    snap_doc.add_argument("--api", default=DEFAULT_API)
    snap_doc.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    snap_doc.add_argument("--json", action="store_true")

    snap_obs = sub.add_parser("snapshot-observations")
    snap_obs.add_argument("--bank", required=True)
    snap_obs.add_argument("--memory-id", required=True)
    snap_obs.add_argument("--case-id", required=True)
    snap_obs.add_argument("--reason", default="")
    snap_obs.add_argument("--api", default=DEFAULT_API)
    snap_obs.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    snap_obs.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument("--case-dir", type=Path)
    verify.add_argument("--case-id")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.add_argument("--json", action="store_true")

    ls = sub.add_parser("list")
    ls.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ls.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "snapshot-document":
        res = snapshot_document(bank=args.bank, document_id=args.document_id, case_id=args.case_id, output_root=args.output_root, api=args.api, reason=args.reason)
    elif args.cmd == "snapshot-observations":
        res = snapshot_observations_for_memory(bank=args.bank, memory_id=args.memory_id, case_id=args.case_id, output_root=args.output_root, api=args.api, reason=args.reason)
    elif args.cmd == "verify":
        case_dir = args.case_dir
        if case_dir is None:
            matches = sorted(args.output_root.glob(f"*-{safe_name(args.case_id or '')}")) if args.case_id else []
            if not matches:
                print("case not found", file=sys.stderr)
                return 2
            case_dir = matches[-1]
        res = verify_case(case_dir)
    elif args.cmd == "list":
        res = {"cases": list_cases(args.output_root)}
    else:
        raise AssertionError(args.cmd)

    if getattr(args, "json", False):
        print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(res, ensure_ascii=False, sort_keys=True))
    return 0 if not isinstance(res, dict) or res.get("ok", True) is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
