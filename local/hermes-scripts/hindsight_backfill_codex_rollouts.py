#!/usr/bin/env python3
"""Backfill codex CLI rollouts (direct + kanban reviewer) into hermes session JSON.

Codex rollout JSONL events ("response_item" / payload.type == "message") are
converted into the native hermes session snapshot format that the offline
manifest scanner reads. Only the gap window (2026-08-27 onward) is covered;
per-message ISO timestamps are written so manifest event_date dating works.

Codex instruction/environment scaffolding (AGENTS.md headers, environment_context,
user_instructions) is dropped — it is boilerplate repeated in every rollout and
would poison tags/classification.

Usage:
    python3 hindsight_backfill_codex_rollouts.py --dry-run
    python3 hindsight_backfill_codex_rollouts.py --write
Safety: never overwrites existing session_*.json; DB/input files opened read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
GAP_START = datetime(2026, 8, 27, 0, 0, tzinfo=CST)

# Map rollout sources to the hermes profile whose sessions dir the manifest scans.
# Direct ~/.codex sessions belong to the default profile; kanban reviewer rollouts
# go under a "codex-reviewer" profile dir (scanner discovers any profile with sessions/).
SOURCES = [
    ("default", "/home/wyr/.codex/sessions"),
    ("reviewer", "/home/wyr/.codex-kanban/reviewer/sessions"),
]
# "default" profile sessions live in the MAIN hermes sessions dir, not profiles/default.
OUT_MAIN = Path("/home/wyr/.hermes/sessions")
OUT_BASE = Path("/home/wyr/.hermes/profiles")

# Prefixes that mark boilerplate rather than real user turns.
SKIP_USER_PREFIXES = (
    "# AGENTS.md instructions",
    "<INSTRUCTIONS>",
    "<environment_context>",
    "<user_instructions>",
    "<turn_context>",
)


def iter_rollouts(base: str):
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".jsonl"):
                p = os.path.join(root, f)
                if os.path.getmtime(p) >= GAP_START.timestamp():
                    yield p


def convert_rollout(path: str) -> dict | None:
    messages = []
    session_start = None
    last_ts = None
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        payload = ev.get("payload", ev)
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        text = " ".join(
            c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict)
        ).strip()
        if not text:
            continue
        if role == "user" and any(text.startswith(pfx) for pfx in SKIP_USER_PREFIXES):
            continue
        ts = ev.get("timestamp") or payload.get("timestamp")
        if ts:
            session_start = session_start or ts
            last_ts = ts
        messages.append({"role": role, "content": text, "timestamp": ts or "", "_db_persisted": True})
    if len(messages) < 3:
        return None
    return {
        "session_id": Path(path).stem.replace("rollout-", "codex-"),
        "model": "codex",
        "base_url": "",
        "platform": "backfill-codex",
        "session_start": session_start or "",
        "last_updated": last_ts or "",
        "system_prompt": "",
        "tools": [],
        "message_count": len(messages),
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.write):
        args.dry_run = True

    total_msgs = total_chars = written = skipped = 0
    for label, base in SOURCES:
        if not os.path.isdir(base):
            continue
        for path in iter_rollouts(base):
            snap = convert_rollout(path)
            if not snap:
                continue
            # per-message epoch -> ISO normalization (codex timestamps are ISO already, but normalize anyway)
            n_chars = sum(len(m["content"]) for m in snap["messages"])
            total_msgs += snap["message_count"]
            total_chars += n_chars
            if label == "default":
                profile_dir = OUT_MAIN  # already IS the sessions dir
                out_path = profile_dir / f"session_{snap['session_id']}.json"
            else:
                profile_dir = OUT_BASE / label
                out_path = profile_dir / "sessions" / f"session_{snap['session_id']}.json"
            exists = out_path.exists()
            status = "EXISTS-skip" if exists else ("write" if args.write else "dry")
            print(
                f"{label:15s} {snap['session_id'][:40]:40s} {snap['message_count']:5d} msgs "
                f"{n_chars:8d} chars [{status}]"
            )
            if exists:
                skipped += 1
                continue
            if args.write:
                profile_dir.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
                tmp.chmod(0o600)
                tmp.replace(out_path)
                written += 1
    print(
        f"\nTOTAL: {total_msgs} msgs, {total_chars} chars (~{total_chars // 8000} chunks); "
        f"written={written} skipped_exists={skipped}"
    )
    if args.dry_run and not args.write:
        print("dry-run only — rerun with --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
